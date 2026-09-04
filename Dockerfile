FROM ubuntu:noble

EXPOSE 5432 8888
STOPSIGNAL SIGINT

# Install dependencies
RUN apt update && apt install -y \
        build-essential meson ninja-build sudo locales tzdata procps lsof wget \
        bison flex curl pkg-config cmake llvm clang \
        libicu-dev libreadline-dev libssl-dev liblz4-dev libossp-uuid-dev libzstd-dev zlib1g-dev \
        liburing-dev \
        git vim unzip zstd default-jre tmux \
        python3 python3-venv python3-pip ; \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

# uv manages the PostBOUND Python environment (see tools/setup-py-venv.sh). It provisions its
# own interpreter from .python-version, so the system python3 is only there for helper scripts.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Configure some general settings
ARG USERNAME=postbound
ENV USERNAME=$USERNAME
ENV USER=$USERNAME

ENV LANG=en_US.UTF-8
ENV LC_ALL=C.UTF-8
ARG TIMEZONE=UTC
ENV TZ=$TIMEZONE
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create a new user
WORKDIR /postbound
WORKDIR /pg_lab
RUN useradd -ms /bin/bash $USERNAME ; \
    echo "$USERNAME:$USERNAME" | chpasswd ; \
    usermod -aG sudo $USERNAME ; \
    echo "$USER ALL=(ALL:ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/$USER
USER $USERNAME

# Final container config
WORKDIR /postbound
VOLUME /postbound
VOLUME /pg_lab
COPY tools/docker-entrypoint.sh /docker-entrypoint.sh
CMD ["/docker-entrypoint.sh"]
