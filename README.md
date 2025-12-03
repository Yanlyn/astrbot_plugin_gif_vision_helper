# AstrBot 插件：GIF Vision Helper

  

## 简介

这是一个 AstrBot 插件，用于在图片描述前自动处理 QQ 的 GIF 动图：

把 GIF 拆分成多帧静态图片，再交给多模态模型，让它真正按“动图”来理解，而不是报错。

为了避免重复造轮子，参考了[GitHub - piexian/astrbot\_plugin\_gif\_to\_video: GIF转视频分析插件，自动为默认服务商或手动指定的服务商启用GIF转视频避免报错。](https://github.com/piexian/astrbot_plugin_gif_to_video)的部分代码段，特此感谢。

  

## 功能

- 自动识别 AstrBot 本地缓存中的 GIF 文件（按文件头魔数判断，不依赖后缀名）

- 将 GIF 抽取为 3～6 帧代表性画面，并统一压缩到合适分辨率

- 覆盖原始 image_urls，将多帧作为多图输入交给多模态模型（如 Qwen-VL / Qwen3-VL）

- 自动在本次对话的 prompt 前插入系统提示，引导模型按“动图多帧”来理解内容

- 静态图片完全不受影响，不会被改写

  

## 效果示例

- 原本：发送 GIF 后，多模态模型可能报错：

  - `image: unknown format`

- 安装本插件后：

  - 模型会按时间顺序描述 GIF 中的动作、表情变化和场景细节

  

## 注意事项

- 本插件只处理 AstrBot 已下载到本地 temp 目录的图片消息

- 大体积 / 高帧数的 GIF 处理会稍慢一些（解码 + 抽帧是必需开销）

- 需要多模态模型支持多图输入，并通过 OpenAI 兼容接口接入 AstrBot

- 依赖 Pillow 库进行 GIF 解码和图像缩放

  

## 安装及使用

1. 将本插件目录放到 AstrBot 的 `data/plugins` 目录，例如：

  

```

AstrBot/

  data/

    plugins/

      gif_vision_helper/

        main.py

        metadata.yaml

        README.md

```

  

2. 确保 `metadata.yaml` 配置正确。

3. 重启 AstrBot 或在面板中重载插件。

4. 在控制台中看到类似日志，说明插件已生效：

```
[astrbot_plugin_gif_vision_helper] 插件已初始化（v0.5，多帧抽样 + 提示注入 + 临时文件管理）
```

典型日志（已成功处理 GIF 的情况）：

  ```
[astrbot_plugin_gif_vision_helper] on_llm_request 捕获到图片请求，尝试处理 GIF: /AstrBot/data/temp/1764734982_33f14bb1.jpg

 [astrbot_plugin_gif_vision_helper] GIF 已成功拆分为 6 帧 JPEG，第 1 帧覆盖原路径，其余帧追加到 image_urls。
  ```

## 配置

- 当前版本无需额外配置，安装即用。

- 如需调整抽帧数量或最大分辨率，可直接修改 `main.py` 顶部的相关常量（例如最长边 768，预览帧数 3～6）。

  

## 思路

- YanL

  

## 编写和修正

- ChatGPT