FROM continuumio/anaconda3

MAINTAINER Colm Ryan <cryan@bbn.com>

RUN apt-get update && apt-get install -y \
 git \
 unzip

RUN pip install pyvisa

#We'd like to use wget but there a bug where --no-check-certificate doesn't work (https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=686837)
RUN curl --insecure -o /tmp/ca.zip https://bua.bbn.com/trac/bua-sysadmin/raw-attachment/wiki/HowToInstallCACerts/ca.zip \
 && unzip tmp/ca.zip -d /tmp \
 && mv /tmp/CA/bbnca.pem /usr/local/share/ca-certificates/bbnca.crt \
 && update-ca-certificates
