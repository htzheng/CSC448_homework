# Do Not Remove This
source /usr/staff/lib/defaults/system.login

# Start up Sun X/NeWS server for display postscript support for answerbook.
# "xnews" needs $OPENWINHOME set and needs /usr/openwin/lib on $LD_LIBRARY_PATH.
# "xnews -favorstatic" improves its colormap handling.
if (`tty` == "/dev/console") then
	echo -n 'Hit ^C in 5 seconds to avoid X...'
	sleep 3
	xinit -- /usr/openwin/bin/xnews -favorstatic
	clear
	echo -n 'Hit ^C in 5 seconds to _not_ logout...'
	sleep 5
	logout
endif
bash
