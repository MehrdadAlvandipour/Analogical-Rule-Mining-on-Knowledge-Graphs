import shlex,subprocess


def file_len(fname):
	cmd = f'wc -l "{fname}"'
	return int(run_proc(cmd)[1].split()[0])

def run_proc(cmd,real_time_output = False, cwd=None):
	'''Run the command _cmd_, wait for it to finish and return its (returncode , stdout&stderr).

	real_time_output=True --> prints the output of the command in real-time. Useful for long processes

	WARNING: It does'nt check if the command fails. Check returncode if necc..
	'''
	#cmd = "java -jar /Users/mehrdadalvandipour/MyDir/X/AMIE/ApplyAMIERules.jar /Users/mehrdadalvandipour/MyDir/X/rules/FB15K237_rules_100p_2020-05-31_18-08-19.txt /Users/mehrdadalvandipour/MyDir/X/datasets/FB15K237/train_dist_100p_2020-05-31_18-08-19.txt /Users/mehrdadalvandipour/MyDir/X/datasets/FB15K237/test.txt /Users/mehrdadalvandipour/MyDir/X/datasets/FB15K237/valid.txt /Users/mehrdadalvandipour/MyDir/X/evaluation/FB15K237_eval_100p_2020-05-31_18-08-19.txt"
	args = shlex.split(cmd)
	
	if real_time_output:
		p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
		while p.poll() is None:
		    l = p.stdout.readline().rstrip()
		    print(l)
		print(p.stdout.read())
	else:
		p = subprocess.run(args,stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, text=True)

	return (p.returncode,p.stdout)