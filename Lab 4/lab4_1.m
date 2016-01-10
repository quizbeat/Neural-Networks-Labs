clear;

%���� �������� ��� ���������� ������������� ��������� �����

%�������� ��������� ��������� �����
X = [0 1.5; 0 1.5];
clusters = 8;
points = 10;
deviation = 0.01;
P = nngenc(X, clusters, points, deviation);
figure;
% 80 �����, ������� ���������� ������������� � 8 ���������
plot(P(1,:),P(2,:),'ob',...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','c',...
                'MarkerSize',7)
title('80 ����� (��������������� � 8 ���������)');
xlabel('��� X');
ylabel('��� Y');            
grid on;
%%
net = competlayer(8); %������ ����
net = configure(net, P);

net.trainParam.epochs = 50;
net = train(net,P);

display(net);
view(net);
%��������� ��������� ���� � ����������� ��������
display('����� ������������� ����:');display(net.initFcn);
display('������������� ������� �������� ����:');
display(net.layers{1}.transferFcn);
display('����� �������� � ������� ����:');display(net.layers{1}.size);
%��������� ��������:
display(net.trainParam.epochs);
display(net.trainParam.show);
display(net.trainParam.showCommandLine);
display(net.trainParam.showWindow);
display(net.trainParam.time);
%%
w = net.IW{1,1};  % ������ ������������� ��� �������������� ���� ����� ��������

display('������ ������������� ����:');
display(w);

%����� 5 ��������� ���������� ������� ��������
Pdop5(1:2, 1:5)= 0;
for i = 1 : 2
    for j = 1 : 5
        Pdop5(i,j)=rand; 
    end
end

display('���� �������������� �����:');
display(Pdop5);
display('�������� ���� �������������� �����:');
display(vec2ind(sim(net,Pdop5)));

figure;
plot(P(1,:),P(2,:),'ob',...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','c',...
                'MarkerSize',7);
hold on;
plot(w(:,1),w(:,2),'^r',...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','r',...
                'MarkerSize',7);
hold on;
plot(Pdop5(1,:),Pdop5(2,:),'rs',...                %���������� ����������� 5 �������������� ��������� �����
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','y',...
                'MarkerSize',7);    
title('������ �� 80 ����� � 5 �������������� ��������� �����');
xlabel('��� X');
ylabel('��� Y');     
legend('80 �������� �����','������ ������������� ����', '�������������� �����', -1);
grid on;

