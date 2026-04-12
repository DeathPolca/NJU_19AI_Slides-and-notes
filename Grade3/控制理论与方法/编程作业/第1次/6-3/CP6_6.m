%(a)
K=[0:1:5];
len=length(K);
num=[1];
for i=1:len
    den=[1 5 K(i)-3 K(i)];
    sys1=tf(num,den);
    sys=feedback(sys1,[1]);
    p(:,i)=pole(sys);
end
plot(real(p),imag(p),'x'),grid
text(0,0.2,'K=0');text(-0.2,1.2,'K=5')

%(b)  计算结果是K>4

%(c)
den=[1 5 1 4];
sys=feedback(tf(num,den),[1]);
p=pole(sys);