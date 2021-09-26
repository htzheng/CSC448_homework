# PATH addition example
setenv PATH "$PATH":"$HOME/bin"

# CUPS print service (allows applications to autodiscover network print queues)
setenv CUPS_SERVER print.csug.rochester.edu

# set default printer
setenv LPDEST inner

# set default editor
setenv EDITOR emacs


# add all INTERACTIVE commands here, specifically, any stty or prompt setting
if ($?prompt) then
        set history = 100

        alias ll ls -l
endif

