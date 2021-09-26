export LANG=en_US.UTF-8
export UNSUPPORTED=

export PS1="\[\e]0;\h:\w\007\][\w]# "
export EDITOR=gvim
export EXINIT='set ts=4|set encoding=utf-8'
export CVS_RSH=ssh

export LESSCHARSET=utf-8

export LS_COLORS='no=00:fi=00:di=00;34:ln=00;36:pi=40;33:so=00;35:bd=40;33;01:cd=40;33;01:or=01;05;37;41:mi=01;05;37;41:ex=00;32:*.cmd=00;32:*.exe=00;32:*.com=00;32:*.btm=00;32:*.bat=00;32:*.sh=00;32:*.csh=00;32:*.tar=00;31:*.tgz=00;31:*.arj=00;31:*.taz=00;31:*.lzh=00;31:*.zip=00;31:*.z=00;31:*.Z=00;31:*.gz=00;31:*.bz2=00;31:*.bz=00;31:*.tz=00;31:*.rpm=00;31:*.cpio=00;31:*.jpg=00;35:*.gif=00;35:*.bmp=00;35:*.xbm=00;35:*.xpm=00;35:*.png=00;35:*.tif=00;35:'

export MAILCHECK=-1

ulimit unlimited

alias cp='cp -i'
alias ls='ls --color -F -h'
alias df='df -h'
alias du='du -h'
alias grep='grep --color'
alias egrep='egrep --color'
alias less='less -e -X -R -f -F'
alias more=less
alias ssh='ssh -X'
alias vim='gvim'
alias vi='gvim'

export POS_TRAIN='/u/cs248/data/pos/train'
export POS_TEST='/u/cs248/data/pos/test'

if [ -f /usr/bin/xemacs ]; then
  alias em='xemacs -nw'
else
  alias em='emacs -nw'
fi

cd ~
