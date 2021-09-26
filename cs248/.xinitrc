#! /bin/sh

xrdb -merge $HOME/.Xdefaults

xsetroot -solid steelblue

# Uncomment this to modify mouse speed settings
#xset m 4 2 b 1 0 10 &

# Uncomment this if want to disable the Lock key
#xmodmap -e 'clear Lock'

(twm; kill -9 $$) &

emacs -geometry =80x40+515+35 &
xterm -geometry 80x24+0+36 &
xterm -geometry 80x24+0+431 &

xclock -analog -geometry 65x65-0+0 &
xbiff -geometry 65x65-0+77 &

exec xterm -C -n "Console" -T "Console" -geometry 80x8-0-0
