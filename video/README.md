# RTSP到Web流媒体转换器

这是一个使用FFmpeg将RTSP流转换为Web前端可播放的流媒体解决方案。

## 功能特性

- 🔄 将RTSP流实时转换为HLS和DASH格式
- 🌐 内置Web服务器，提供流媒体服务和播放器界面
- 📱 支持多种设备和浏览器
- ⚙️ 灵活的配置选项
- 🔄 自动重连机制
- 📊 实时状态监控

## 系统要求

- Python 3.7+
- FFmpeg（必须安装并添加到PATH环境变量）
- 支持RTSP协议的摄像头或流媒体源

## 安装步骤

### 1. 安装FFmpeg

**Windows:**
1. 下载FFmpeg: https://ffmpeg.org/download.html
2. 解压到某个目录（如 `C:\ffmpeg`）
3. 将 `C:\ffmpeg\bin` 添加到系统PATH环境变量
4. 验证安装：在命令行运行 `ffmpeg -version`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 2. 配置RTSP源

编辑 `ffmpeg_config.json` 文件，修改RTSP URL：

```json
{
  "rtsp_config": {
    "url": "rtsp://username:password@camera_ip:port/stream"
  }
}
```

## 使用方法

### 快速启动

1. 双击运行 `start_streaming.bat`（Windows）
2. 或在命令行运行：
   ```bash
   python rtsp_to_web.py
   ```

3. 打开浏览器访问：`http://localhost:8080`

### 配置选项

主要配置文件 `ffmpeg_config.json`：

```json
{
  "ffmpeg_path": "ffmpeg",
  "rtsp_config": {
    "url": "rtsp://admin:password@192.168.1.100",
    "timeout": 30,
    "reconnect_delay": 5,
    "max_reconnect_attempts": 10
  },
  "output_formats": {
    "hls": {
      "enabled": true,
      "segment_time": 4,
      "segment_list_size": 6,
      "output_dir": "./hls_output",
      "playlist": "stream.m3u8"
    },
    "dash": {
      "enabled": true,
      "segment_duration": 4,
      "output_dir": "./dash_output",
      "manifest": "stream.mpd"
    }
  },
  "stream_settings": {
    "video_codec": "libx264",
    "audio_codec": "aac",
    "bitrate": "2000k",
    "resolution": "1280x720",
    "framerate": 25,
    "preset": "ultrafast",
    "tune": "zerolatency"
  },
  "web_server": {
    "port": 8080,
    "static_dir": "./public"
  }
}
```

### 流媒体URL

- **HLS流**: `http://localhost:8080/hls/stream.m3u8`
- **DASH流**: `http://localhost:8080/dash/stream.mpd`

## 网页播放器

系统提供内置的Web播放器，支持：

- HLS格式播放
- DASH格式播放
- 实时状态监控
- 流媒体控制（开始/停止）
- 多标签页切换

## 故障排除

### 常见问题

1. **FFmpeg未找到**
   - 确保FFmpeg已正确安装
   - 检查PATH环境变量设置
   - 在配置文件中指定FFmpeg完整路径

2. **RTSP连接失败**
   - 检查RTSP URL是否正确
   - 确认网络连接正常
   - 验证摄像头凭据
   - 检查防火墙设置

3. **端口被占用**
   - 修改配置文件中的端口号
   - 检查是否有其他程序占用端口

4. **播放延迟较高**
   - 调整 `segment_time` 参数
   - 使用 `ultrafast` 预设
   - 降低视频比特率

### 调试模式

查看详细日志：
```bash
python rtsp_to_web.py
```

日志文件：`rtsp_stream.log`

## 性能优化

### 低延迟设置
```json
{
  "stream_settings": {
    "preset": "ultrafast",
    "tune": "zerolatency",
    "segment_time": 2
  }
}
```

### 网络优化
```json
{
  "rtsp_config": {
    "timeout": 10,
    "reconnect_delay": 2
  }
}
```

### 质量优化
```json
{
  "stream_settings": {
    "bitrate": "4000k",
    "resolution": "1920x1080",
    "preset": "fast"
  }
}
```

## 安全考虑

- 不要在配置文件中暴露敏感信息
- 使用强密码保护RTSP流
- 考虑添加身份验证机制
- 定期更新FFmpeg版本

## 许可证

本项目基于MIT许可证开源。

## 支持

如有问题或建议，请提交Issue或Pull Request。