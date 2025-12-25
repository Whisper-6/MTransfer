#!/bin/bash
# 取消 V2Ray 代理设置脚本

# 取消所有代理相关环境变量
unset http_proxy
unset https_proxy
unset ftp_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY
unset ALL_PROXY
unset all_proxy
unset no_proxy
unset NO_PROXY

echo "✓ 代理设置已清除"
echo ""
echo "使用方式："
echo "  source /cephfs/shared/jiarui/v2ray/unset_network.sh"

