%(a)
num1=[1 1];
den1=[1 2];
sys1=tf(num1,den1);

fdnum=[1];
fdden=[1 1];
fd=tf(fdnum,fdden);

sys=feedback(sys1,fd,-1);

%(b)
pzmap(sys);
p=pole(sys);
z=zero(sys);

%(c) 可以对消
syss=minreal(sys);

%(d) 零极点对消可以化简传递函数