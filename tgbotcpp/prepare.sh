#!/usr/bin/bash

mkdir sysroots && cd sysroots && \
sudo debootstrap --arch arm64 bookworm debian_arm64_bookworm http://deb.debian.org/debian/ && \
#sudo mount --bind /proc debian_arm64_bookworm/proc && \
#sudo mount --bind /sys debian_arm64_bookworm/sys && \
#sudo mount --bind /dev debian_arm64_bookworm/dev && \
sudo chroot debian_arm64_bookworm /bin/bash -c "apt update && apt install -y apt-utils && apt show linux-libc-dev libc6"