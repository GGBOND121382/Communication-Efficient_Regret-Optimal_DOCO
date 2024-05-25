FROM python:slim

COPY . DOCO

RUN cd DOCO

RUN python -m pip install numpy scikit-learn scipy matplotlib wget