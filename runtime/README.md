# 本地 FFmpeg Runtime

`runtime/bin/` 不提交到 Git。项目默认优先使用：

```text
runtime/bin/ffmpeg
runtime/bin/ffprobe
```

## 当前本地构建

- FFmpeg 7.1.1
- LGPL v2.1+：配置包含 `--disable-gpl --disable-nonfree --disable-version3`
- 无外部编解码库
- 支持解码：H.264、HEVC、VP8、VP9、AV1、MPEG-4、MJPEG
- 支持 image2 输出和 FPS/裁剪/缩放滤镜

## 自行构建

```bash
curl -L -o ffmpeg.tar.xz https://ffmpeg.org/releases/ffmpeg-7.1.1.tar.xz
tar -xf ffmpeg.tar.xz
cd ffmpeg-7.1.1

./configure \
  --prefix=/tmp/ffmpeg-lgpl/prefix \
  --disable-gpl --disable-nonfree --disable-version3 \
  --disable-doc --disable-network --disable-autodetect --disable-everything \
  --enable-ffmpeg --enable-ffprobe \
  --enable-avcodec --enable-avformat --enable-avfilter --enable-swscale \
  --enable-decoder=h264,hevc,vp8,vp9,av1,mpeg4,mjpeg \
  --enable-parser=h264,hevc,vp8,vp9,av1,mpeg4video \
  --enable-demuxer=mov,matroska,image2 \
  --enable-muxer=image2 --enable-encoder=mjpeg \
  --enable-filter=fps,scale,crop,format \
  --enable-protocol=file

make -j"$(sysctl -n hw.ncpu)"
make install
```

将 `prefix/bin/ffmpeg`、`prefix/bin/ffprobe` 和 FFmpeg 源码中的
`COPYING.LGPLv2.1` 放入 `runtime/bin/`。

## 许可证核对

```bash
runtime/bin/ffmpeg -L | grep -i "Lesser General Public"
runtime/bin/ffmpeg -buildconf | grep -E -- "--enable-gpl|--enable-nonfree"
```

第二条命令应无输出。不要默认使用启用了 GPL 或 nonfree 组件的构建进行分发。

## 配置覆盖

可在 `.env` 指定：

```dotenv
FFMPEG_BINARY=/absolute/path/to/ffmpeg
FFPROBE_BINARY=/absolute/path/to/ffprobe
```
