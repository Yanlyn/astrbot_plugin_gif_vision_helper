from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import List

from PIL import Image

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp


@register(
    "astrbot_plugin_gif_vision_helper",
    "YanL",
    "将 QQ 的 GIF 动图拆分为多帧静态图，并注入提示，方便多模态模型理解",
    "0.5.0",
)
class GifVisionHelper(Star):
    """
    v0.5 功能概览：

    - 仅在检测到“本地是真 GIF 文件”时触发，静态图完全旁路。
    - 使用 Pillow 从 GIF 中抽帧采样，生成 3~6 张代表性帧。
    - 按最长边 768px 进行等比缩放，控制显存 / 带宽占用。
    - 将多帧保存为本地 JPEG 文件，并写回 req.image_urls，让多模态模型走多图输入。
    - 在 prompt 前方注入系统提示，引导模型综合多帧理解动图语义。
    - 记录临时帧文件，插件终止时清理 + 简单 TTL（24h）过期清理。
    """

    PLUGIN_NAME = "astrbot_plugin_gif_vision_helper"

    def __init__(self, context: Context):
        super().__init__(context)

        # 临时帧管理
        self._temp_files: set[Path] = set()
        self._temp_files_lock = threading.Lock()
        self._cache_ttl = 86400  # 24h TTL，用于过期临时帧清理

        # 抽帧 / 尺寸策略
        self.max_preview_frames = 6   # 上限帧数
        self.min_preview_frames = 3   # 下限帧数
        self.default_preview_frames = 5
        self.max_side = 768  # 统一最长边限制

    async def initialize(self):
        logger.info(
            f"[{self.PLUGIN_NAME}] 插件已初始化（v0.5，多帧抽样 + 提示注入 + 临时文件管理）"
        )

    # ---------- 基础工具函数 ----------

    def _register_temp_file(self, file_path: Path) -> None:
        """记录临时文件，便于后续统一清理"""
        with self._temp_files_lock:
            self._temp_files.add(file_path)

    def _cleanup_temp_files(self) -> None:
        """插件卸载时清理所有已记录的临时文件"""
        with self._temp_files_lock:
            for temp in list(self._temp_files):
                try:
                    if temp.exists():
                        temp.unlink()
                    parent = temp.parent
                    # 如果目录已经空了顺便清一下
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                except Exception as e:
                    logger.warning(
                        f"[{self.PLUGIN_NAME}] 清理临时文件失败: {temp} - {e}",
                        exc_info=True,
                    )
                finally:
                    self._temp_files.discard(temp)

    def _cleanup_expired_cache(self) -> None:
        """
        简单 TTL 清理：删除记录中 mtime 超过 _cache_ttl 的临时帧
        （每次 on_llm_request 调用时顺带做一下，避免长期堆积）
        """
        now = time.time()
        with self._temp_files_lock:
            for temp in list(self._temp_files):
                try:
                    if (not temp.exists()) or now - temp.stat().st_mtime > self._cache_ttl:
                        if temp.exists():
                            temp.unlink()
                        self._temp_files.discard(temp)
                except FileNotFoundError:
                    self._temp_files.discard(temp)
                except Exception as e:
                    logger.warning(
                        f"[{self.PLUGIN_NAME}] TTL 清理失败: {temp} - {e}",
                        exc_info=True,
                    )

    @staticmethod
    def _is_gif_bytes(head: bytes) -> bool:
        """根据本地文件魔数判断是否为 GIF"""
        return head.startswith(b"GIF87a") or head.startswith(b"GIF89a")

    def _resize_frame(self, frame: Image.Image) -> Image.Image:
        """按最长边 max_side 等比缩放一帧"""
        w, h = frame.size
        longest = max(w, h)
        if longest <= 0 or longest <= self.max_side:
            return frame

        scale = self.max_side / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        try:
            frame = frame.resize(new_size, Image.LANCZOS)
        except Exception:
            # 某些环境可能没有 LANCZOS，退回默认
            frame = frame.resize(new_size)
        return frame

    @staticmethod
    def _evenly_sample_indices(total: int, target: int) -> List[int]:
        """
        从 [0, total-1] 里等间隔采样 target 个索引。
        确保首尾一定包含（类似你给的 0% ~ 100% 分布）。
        """
        if total <= 0:
            return []
        if target >= total:
            return list(range(total))

        # 使用 (i/(target-1)) * (total-1) 这样的等间隔方案
        indices = []
        for i in range(target):
            pos = round(i * (total - 1) / (target - 1))
            if pos not in indices:
                indices.append(pos)
        indices.sort()
        return indices

    def _decide_preview_frame_count(self, frame_count: int, file_size_kb: float) -> int:
        """
        根据 GIF 总帧数 + 文件体积动态决定抽样帧数。
        只是一个经验规则，可以按你之后实际体验继续微调。
        """
        # 小而精：体积小、帧数也不多，多给一点上下文
        if frame_count <= 20 and file_size_kb <= 512:
            target = self.max_preview_frames  # 6

        # 中等：默认 4~5 帧
        elif frame_count <= 80 and file_size_kb <= 2048:
            target = self.default_preview_frames  # 5

        # 大而长：非常长或非常大，降到 3~4 帧，避免爆显存
        else:
            target = max(self.min_preview_frames, self.default_preview_frames - 1)  # 至少 3

        # 不超过真实帧数
        target = max(self.min_preview_frames if frame_count > 1 else 1, min(target, frame_count))
        return target

    def _extract_sampled_frames(self, gif_path: Path) -> List[Image.Image]:
        """
        从本地 GIF 文件抽取若干代表性帧（已经做尺寸压缩）。

        返回值：若干 Pillow Image 对象（RGB，最长边不超过 max_side）
        """
        file_size_kb = 0.0
        try:
            if gif_path.exists():
                file_size_kb = gif_path.stat().st_size / 1024.0
        except Exception:
            pass

        frames: List[Image.Image] = []
        with Image.open(str(gif_path)) as img:
            if not getattr(img, "is_animated", False):
                # 退化为单张图
                frame = img.convert("RGB")
                frame = self._resize_frame(frame)
                frames.append(frame)
                return frames

            total_frames = getattr(img, "n_frames", 1) or 1
            target = self._decide_preview_frame_count(total_frames, file_size_kb)
            indices = self._evenly_sample_indices(total_frames, target)

            for idx in indices:
                try:
                    img.seek(idx)
                    frame = img.convert("RGB")
                    frame = self._resize_frame(frame)
                    frames.append(frame.copy())
                except EOFError:
                    logger.warning(
                        f"[{self.PLUGIN_NAME}] 抽帧时出现 EOFError，frame_idx={idx}, total={total_frames}"
                    )
                except Exception as e:
                    logger.warning(
                        f"[{self.PLUGIN_NAME}] 抽帧失败 frame_idx={idx}: {e}",
                        exc_info=True,
                    )

        if not frames:
            # 极端情况下至少返回一帧
            with Image.open(str(gif_path)) as img:
                frame = img.convert("RGB")
                frame = self._resize_frame(frame)
                frames.append(frame)
        return frames

    def _inject_preview_hint(self, prompt: str | None, frame_count: int) -> str:
        """
        在 prompt 中注入说明，提醒 LLM 这一次看到的是“GIF 多帧静态预览”。
        """
        if frame_count > 1:
            hint = (
                f"[系统提示] 检测到一张 GIF 动图，本插件已从中抽取 {frame_count} 帧静态图片。"
                f" 请综合所有帧理解整段动画的内容、时序和情绪变化。"
            )
        else:
            hint = (
                "[系统提示] 检测到 GIF 动图，但仅成功提取到 1 帧静态图片；"
                "请结合这帧画面和聊天上下文进行理解。"
            )

        prompt = prompt or ""
        if hint in prompt:
            return prompt
        return f"{hint}\n{prompt}" if prompt else hint

    def _convert_gif_to_multi_jpeg(self, gif_path: Path, req) -> None:
        """
        核心同步逻辑：
        - 抽帧 + 缩放 -> 得到若干 Pillow Image
        - 覆盖原 image_urls[0] 为首帧 JPEG
        - 附加额外帧到 image_urls，其余字段不变
        - 注入系统提示到 prompt
        """
        # 安全判断：确认真的是 GIF
        try:
            with open(gif_path, "rb") as f:
                head = f.read(8)
        except FileNotFoundError:
            logger.warning(
                f"[{self.PLUGIN_NAME}] 本地 GIF 路径不存在，跳过处理: {gif_path}"
            )
            return
        except Exception as e:
            logger.warning(
                f"[{self.PLUGIN_NAME}] 读取 GIF 头部失败: {gif_path} - {e}",
                exc_info=True,
            )
            return

        if not self._is_gif_bytes(head):
            # 头部不是 GIF，直接忽略
            logger.debug(
                f"[{self.PLUGIN_NAME}] 本地文件魔数非 GIF，直接旁路: {gif_path}"
            )
            return

        # 抽样出多帧
        frames = self._extract_sampled_frames(gif_path)
        if not frames:
            logger.warning(
                f"[{self.PLUGIN_NAME}] 从 GIF 中未能成功抽出任何帧，保持原始行为。"
            )
            return

        # 确保 req.image_urls 存在
        if not hasattr(req, "image_urls") or req.image_urls is None:
            req.image_urls = []

        # 原始 image_urls
        old_urls: List[str] = list(req.image_urls)

        new_urls: List[str] = []
        parent_dir = gif_path.parent
        stem = gif_path.stem  # 通常是 AstrBot 自动命名的 temp 文件名

        for idx, frame in enumerate(frames):
            if idx == 0:
                # 覆盖原路径：让之前会报错的“GIF 文件”摇身一变，成为首帧 JPEG
                out_path = gif_path
            else:
                # 额外帧：追加新的临时文件
                out_path = parent_dir / f"{stem}_fv{idx}.jpg"

            try:
                frame.save(str(out_path), format="JPEG", quality=92, optimize=True)
            except Exception as e:
                logger.warning(
                    f"[{self.PLUGIN_NAME}] 写入帧失败 idx={idx}, path={out_path}: {e}",
                    exc_info=True,
                )
                continue

            if idx > 0:
                # 首帧就是原 temp 文件，不额外登记；其他帧需要登记以便后续清理
                self._register_temp_file(out_path)

            new_urls.append(str(out_path))

        if not new_urls:
            logger.warning(
                f"[{self.PLUGIN_NAME}] 没有成功生成任何 JPEG 帧，保留原始 image_urls。"
            )
            return

        # 将新帧列表写回 image_urls，保留原本的“多图场景”语义
        if old_urls:
            # 假设当前 GIF 对应第 0 个索引（AstrBot 当前实现里图片描述一般就是这样）
            rest = old_urls[1:]
            req.image_urls = new_urls + rest
        else:
            req.image_urls = new_urls

        # 注入系统提示到 prompt
        current_prompt = getattr(req, "prompt", "")
        req.prompt = self._inject_preview_hint(current_prompt, len(new_urls))

        logger.info(
            f"[{self.PLUGIN_NAME}] GIF 已成功拆分为 {len(new_urls)} 帧 JPEG，"
            f"第 1 帧覆盖原路径，其余帧追加到 image_urls。"
        )

    # ---------- LLM 请求钩子 ----------

    @filter.on_llm_request(priority=100)
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """
        关键 hook：
        - 检测当前消息是否包含 QQ 图片（不区分静态/动图）
        - 定位对应的本地 temp 文件路径（req.image_urls[0]）
        - 若本地魔数显示为 GIF -> 抽帧 + 多图注入 + 提示注入
        """
        # 顺便做一轮 TTL 清理
        self._cleanup_expired_cache()

        # 保护性判空
        msg_obj = getattr(event, "message_obj", None)
        if msg_obj is None:
            return

        msg_chain = getattr(msg_obj, "message", None)
        if not isinstance(msg_chain, list) or not msg_chain:
            return

        # 当前场景主要考虑“单图 + 描述”：只要发现有 Image 组件就继续
        has_image = any(isinstance(c, Comp.Image) for c in msg_chain)
        if not has_image:
            return

        # ProviderRequest 里应该已经填充了本地下载后的 temp 路径
        image_urls = getattr(req, "image_urls", None)
        if not image_urls:
            logger.debug(
                f"[{self.PLUGIN_NAME}] 发现图片组件，但 ProviderRequest.image_urls 为空，跳过。"
            )
            return

        # 当前版本假定“本轮描述的主图”对应第 0 个索引
        gif_path = Path(image_urls[0])

        logger.info(
            f"[{self.PLUGIN_NAME}] on_llm_request 捕获到图片请求，尝试处理 GIF: {gif_path}"
        )

        # 在独立线程里做所有 IO / CPU 操作，避免阻塞事件循环
        await asyncio.to_thread(self._convert_gif_to_multi_jpeg, gif_path, req)

    async def terminate(self):
        """插件被卸载 / AstrBot 退出时调用"""
        logger.info(f"[{self.PLUGIN_NAME}] 插件终止，开始清理临时文件")
        self._cleanup_temp_files()
