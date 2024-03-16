FROM python:3.11

WORKDIR /fl_attacks

COPY ./ ./

RUN make install

CMD [ "make", "adult_back_krum" ]