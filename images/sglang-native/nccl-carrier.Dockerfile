FROM busybox:1.37.0@sha256:9db7b59979c38555a39def84a31fb98b5296952f9e3afd4f6f11f05b07adfab0
LABEL org.opencontainers.image.title="Experimental NCCL 2.30.7 switchless-ring carrier"
COPY runtime/ /opt/nccl-mesh/
ENTRYPOINT ["/bin/sh"]
