# I use ARG DEBIAN_FRONTEND=noninteractive so that installation of tzdata does not fail waiting for user input
#       https://stackoverflow.com/questions/67452096/docker-build-hangs-based-on-order-of-install
# I set SHELL to bash with "-l" to ensure it sources .bash_profile for pyenv init
#       https://stackoverflow.com/questions/55206227/why-bashrc-is-not-executed-when-run-docker-container
# I install VSCode extensions in advance to (slightly) speed things up if using VSCode
#       https://stackoverflow.com/questions/63354237/how-to-install-vs-code-extensions-in-a-dockerfile
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update; apt-get install vim curl -y

# python
RUN apt-get install python3-pip -y

# pyenv dependencies
RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git -y
RUN curl https://pyenv.run | bash
RUN echo "export PYENV_ROOT=\"\$HOME/.pyenv\"\n[[ -d \$PYENV_ROOT/bin ]] && export PATH=\"\$PYENV_ROOT/bin:\$PATH\"\neval \"\$(pyenv init -)\"" >> ~/.bash_profile
RUN echo "export PYENV_ROOT=\"\$HOME/.pyenv\"\n[[ -d \$PYENV_ROOT/bin ]] && export PATH=\"\$PYENV_ROOT/bin:\$PATH\"\neval \"\$(pyenv init -)\"" >> ~/.bashrc
SHELL ["/bin/bash", "-l", "-c"]
RUN source ~/.bash_profile

# VSCode
RUN curl -fsSL https://code-server.dev/install.sh | sh
RUN code-server --install-extension ms-python.python
RUN pyenv install 3.11.7

# install jupyter lab
COPY jupyter-requirements.txt /jupyter-requirements.txt
RUN pyenv virtualenv 3.11.7 jupyter
RUN pyenv activate jupyter; python3 -m pip install --upgrade pip; python3 -m pip install -r jupyter-requirements.txt; pyenv deactivate jupyter

# install ipython kernel
COPY requirements.txt /requirements.txt
RUN pyenv virtualenv 3.11.7 udacity-customer-segments; pyenv activate udacity-customer-segments; python3 -m pip install --upgrade pip; python3 -m pip install -r requirements.txt; python3 -m ipykernel install --user --name udacity-customer-segments; pyenv deactivate udacity-customer-segments

# command (with optionally specified port)
ENV port 8889
CMD bash -c "source ~/.bash_profile && source /etc/bash.bashrc && export SHELL=/bin/bash && pyenv activate jupyter; python3 -m jupyter lab --port=${port} --notebook-dir=/work/ --ip 0.0.0.0 --no-browser --allow-root"