FROM busybox:1.37.0@sha256:9db7b59979c38555a39def84a31fb98b5296952f9e3afd4f6f11f05b07adfab0

LABEL org.opencontainers.image.title="spark-manage NCCL Mesh runtime" \
      org.opencontainers.image.source="https://github.com/NVIDIA/nccl" \
      dev.spark-manage.nccl.commit="b91894bd5b190c874d98a017f93f5daa515b65d0" \
      dev.spark-manage.mesh.commit="19924dcc7c571d6e260953724d394ae50bad82cf"

COPY runtime/ /opt/nccl-mesh/

ENTRYPOINT ["/bin/sh"]
