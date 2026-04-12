num=[1];
den=[1 8 10 1];
sys=tf(num,den);
rlocus(sys);

%阻尼比为0.707时，K=5.23