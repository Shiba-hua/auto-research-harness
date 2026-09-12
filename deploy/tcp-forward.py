#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把平台代理转发的 0.0.0.0:<PORT> 转给 DSH 的 127.0.0.1:18080。

为什么需要它：DSH 出于安全拒绝直接绑定 0.0.0.0（"会向网络暴露远程代码执行"），
而平台的反向代理（202.112.194.63）只能访问容器内 0.0.0.0 上的服务。
故用一个本机转发器搭桥：对外 0.0.0.0:20294 -> 内 127.0.0.1:18080。

纯 stdlib、无线程池上限、支持长连接与半关闭。
"""
import socket
import sys
import threading

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 20294
TARGET_HOST = "127.0.0.1"
TARGET_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 18080
BUF = 65536


def pipe(src, dst):
    try:
        while True:
            data = src.recv(BUF)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def handle(conn):
    try:
        up = socket.create_connection((TARGET_HOST, TARGET_PORT), timeout=10)
    except OSError as e:
        print("upstream connect failed: %r" % (e,), flush=True)
        conn.close()
        return
    up.settimeout(None)
    conn.settimeout(None)
    t1 = threading.Thread(target=pipe, args=(conn, up), daemon=True)
    t2 = threading.Thread(target=pipe, args=(up, conn), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    for s in (conn, up):
        try:
            s.close()
        except OSError:
            pass


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((LISTEN_HOST, LISTEN_PORT))
    srv.listen(128)
    print("forward %s:%d -> %s:%d" % (LISTEN_HOST, LISTEN_PORT, TARGET_HOST, TARGET_PORT), flush=True)
    while True:
        try:
            conn, addr = srv.accept()
        except OSError:
            continue
        print("conn from %s" % (addr,), flush=True)
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
