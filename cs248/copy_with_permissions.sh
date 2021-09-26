#!/bin/bash
#
# 08/29/03 - copied from /u/myros/commands
#
# script to turn in a CSC 247/447 assignment
# takes a single argument: the name of the directory whose files
# should be made available to the TA
#
# the following lines have to be customized for troi vs. CS machines:

#echo $0;
#echo $1;
#echo $2;


if test $# -ne 4 ; then
    echo "This script takes 4 arguments: the source directory, the destination directory, the new file permissions, and the new directory permissions to be set"
    exit 1
fi

sourced=$1;
targetd=$2;

fperm=$3;
dperm=$4;


if test ! -d $sourced ; then
    echo "$sourced is not a directory"
    exit 1
fi


#echo `pwd`
#echo "mkdir $targetd"


mkdir $targetd
chmod $dperm $targetd
chmod a+x $targetd

#ls -al $targetd

cd $sourced

trap "/bin/rm -r -f $targetd ; exit 1" 1 2 3 15
for file in `du -a | sed -e "s/^.*	\.\/*//" | sort` ; do
    # that produces a recursive list of all files
    # the sort should guarantee that directories come before their contents
    case $file in
        *,v)
            ;;
        *)
            if test -d $file ; then
	       if test -x $file ; then
	          if test -r $file ; then
		      echo "mkdir .../$file"
		      mkdir $targetd/$file
		      chmod $dperm $targetd/$file
		  else
		      echo "ERROR: cannot read $file directory because it does not have read permission for others."
                      exit 1
		  fi
	       else
                   echo "ERROR: cannot read $file directory because it does not have execute permission for others"
	       fi
            else
                # an interesting file
		if test ! -r $file ; then
		    echo "ERROR: I cannot copy $file because I do not have read permission"
		    exit 1
                else
		    echo $file
		    cp ./$file $targetd/$file
		    chmod $fperm $targetd/$file
		    if test -x ./$file ; then
			chmod a+x $targetd/$file
		    fi
		fi
	    fi
    esac
done

exit 0



