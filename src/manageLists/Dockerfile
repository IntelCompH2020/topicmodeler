# # Base image
# FROM python:3.8

# # Directories architecture
# WORKDIR /app
# RUN mkdir -p /data/wordlists

# # Application
# COPY manageLists.py .

# # Execution
# ENTRYPOINT ["python", "manageLists.py", "--path_wordlists", "/data/wordlists"]
# CMD ["-h"]

# Base image
FROM ubuntu:20.04

# Install python
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-distutils \
    python3-pip \
    python3-apt

RUN apt install python-is-python3

# Directories architecture
WORKDIR /app
RUN mkdir -p /data/wordlists

# Application
COPY manageLists.py .

# Execution
# ENTRYPOINT ["bash"]
ENTRYPOINT ["python", "manageLists.py", "--path_wordlists", "/data/wordlists"]
CMD ["-h"]
