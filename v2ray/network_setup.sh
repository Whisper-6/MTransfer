#!/bin/bash
# V2Ray 代理设置脚本
# 根据 config.json: HTTP=10810, SOCKS5=10809

# HTTP/HTTPS/FTP 使用 HTTP 代理协议（端口 10810）
export http_proxy=http://127.0.0.1:10810
export https_proxy=http://127.0.0.1:10810
export ftp_proxy=http://127.0.0.1:10810
export HTTP_PROXY=http://127.0.0.1:10810
export HTTPS_PROXY=http://127.0.0.1:10810
export FTP_PROXY=http://127.0.0.1:10810

# ALL_PROXY 使用 SOCKS5 协议（端口 10809）
export ALL_PROXY=socks5://127.0.0.1:10809
export all_proxy=socks5://127.0.0.1:10809

# 不使用代理的地址
export no_proxy=localhost,127.0.0.0/8,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,10.210.22.10,.sqz.ac.cn,58.32.7.115
export NO_PROXY=localhost,127.0.0.0/8,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,10.210.22.10,.sqz.ac.cn,58.32.7.115

echo "✓ 代理已设置："
echo "  HTTP/HTTPS: http://127.0.0.1:10810"
echo "  SOCKS5:     socks5://127.0.0.1:10809"
echo ""
echo "使用方式："
echo "  source /cephfs/shared/jiarui/v2ray/network_setup.sh"
