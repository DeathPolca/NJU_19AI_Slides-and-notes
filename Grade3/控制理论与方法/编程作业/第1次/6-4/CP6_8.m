%(a) 计算出来的结果是0<K1<20
%(b)
K1=[0:1:30];
num1=[5];
den1=[1 10 0];
sys1=tf(num1,den1);
for i=1:length(K1)
    num2=[2 K1(i)];
    den2=[1 0];
    sys2=tf(num2,den2);
    sys=feedback(sys1,sys2,-1);
    p(:,i)=pole(sys); 
end
plot(real(p),imag(p),'x'),grid
text(0.7,3.8,'K=30');