N=5000;
x=rand(5000,10);
NN_idx=[];% 每个记录最近样本的idx
NN_dis=[];
% 开始计算最近邻；
t1=cputime;
for i = 1:N
    min_dis=10000;
    min_idx=10000;
    for j = 1:N
        if j~=i
            temp_dis=sqrt((x(0*N+i)-x(0*N+j))^2+(x(1*N+i)-x(1*N+j))^2+(x(2*N+i)-x(2*N+j))^2+(x(3*N+i)-x(3*N+j))^2+(x(4*N+i)-x(4*N+j))^2+(x(5*N+i)-x(5*N+j))^2+(x(6*N+i)-x(6*N+j))^2+(x(7*N+i)-x(7*N+j))^2+(x(8*N+i)-x(8*N+j))^2+(x(9*N+i)-x(9*N+j))^2);
            temp_idx=j;
            if temp_dis<min_dis
                min_dis=temp_dis;
                min_idx=temp_idx;
            end
        end
    end
    NN_idx(i)=min_idx; %第i个元素对应的最近邻索引
    NN_dis(i)=min_dis;
end
t2=cputime;
t=t2-t1;