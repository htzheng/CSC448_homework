/*
  Copied from /u/myros/commands
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <pwd.h>
#include <sys/types.h>

#define COMMAND "/u/cs248/copy_with_permissions.sh"

 char *get_user_name(){
	struct passwd *pass  = NULL;
	char *name = NULL;
	uid_t id = getuid();
	pass = getpwuid(id);
	if (pass) {
	    name = pass->pw_name;
	}
	return name;
}

 char *get_eff_user_name(){
	struct passwd *pass  = NULL;
	char *name = NULL;
	uid_t id = geteuid();
	pass = getpwuid(id);
	if (pass) {
	    name = pass->pw_name;
	}
	return name;
}


int main(int argc, char* const argv[], char *envp[]) {   
    //    printf("Real uid: %s effective uid: %s \n", get_user_name(), get_eff_user_name());
    // a little environment cleanup for savety reasons
    setenv("PATH", "/bin:/usr/bin", 1);
    setenv("IFS", " ", 1);
    unsetenv("BASH_ENV");
    unsetenv("CDPATH");


    char** newargv = new char* [argc + 3];
    newargv[0] = argv[0];
    for (int i=1; i < argc; i++) {
	// another safety check: don't try to give me any dangerous chars here
	if (strpbrk(argv[i],"[]\\:;$()|<>&")) {
	    printf("Dangerous characters in an argument name %s - cannot be executed!!!\n", argv[i]);
	    exit(5);
	}
	newargv[i] = argv[i];
    }
    newargv[argc] = "600";
    newargv[argc+1] = "700";
    newargv[argc+2] = NULL;
    
    setreuid(geteuid(), geteuid());
    execv(COMMAND, newargv);
    //   printf("Real uid: %s effective uid: %s \n", get_user_name(), get_eff_user_name());    
}
