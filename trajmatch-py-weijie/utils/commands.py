import os


def clear_path(path):
    print("Removed [%s]:%s" % (path, str(os.listdir(path))))
    os.system("rm -rf %s/*" % path)


def cp_path(path1, path2, clear_path2=True):
    if (clear_path2):
        clear_path(path2)
    os.system("cp -r %s/* %s/" % (path1, path2))


def exec_match_command(opt, java_args=None):
    match_command = \
        'cd %s && mvn compile \
        && mvn exec:java \
         -Dexec.mainClass=algorithm.mapmatching.MapMatchingMain\
         -Dexec.args="%s" \
         -Dexec.cleanupDaemonThreads=false' % (opt.java_src_path,java_args if java_args else opt.java_args)
    os.system(match_command)
