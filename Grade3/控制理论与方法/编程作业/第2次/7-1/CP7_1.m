% (a)
num=[10];
den=[1 14 43 30];
sys=tf(num,den);
rlocus(sys);

% (b)
num=[1 20];
den=[1 4 20];
sys=tf(num,den);
rlocus(sys);

% (c)
num=[1 1 2];
den=[1 6 10 0];
sys=tf(num,den);
rlocus(sys);

% (d)
num=[1 4 6 10 6 4];
den=[1 4 4 1 1 10 1];
sys=tf(num,den);
rlocus(sys);