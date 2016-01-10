clear;

%����� �������� ��� ���������� ������ �� ������� ������ ������������

%����� 20 ��������� ���������� ������� ��������
T(1:2, 1:20)= 0;
for i = 1 : 2
    for j = 1 : 20
        T(i,j)=3.0*rand-1.5; 
    end
end

figure;
hold on;
plot(T(1,:),T(2,:),'-v','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize', 7), grid;
title('����� 20 �������');
xlabel('��� X');
ylabel('��� Y'); 
hold off;

%������ ���� ��������, ������� ������� �������� ���������� �������

net = newsom(T, 20);

view(net);

net.trainParam.epochs = 600; %����� ����
net = train(net,T); %�������� �������� �� ��������� �������

%��������� ��������� ���� � ����������� ��������
display('����� ������������� ����:');display(net.initFcn);
display('������������� ������� �������� ����:');
display(net.layers{1}.transferFcn);
display('����� �������� � ������� ����:');display(net.layers{1}.size);
%��������� ��������:
display(net.trainParam.epochs);
display(net.trainParam.show);
display(net.trainParam.showCommandLine);
display(net.trainParam.time);

%��������� ���������� ������� � ������ ���������, ��������������� �����

figure;
hold on;
plot(T(1,:),T(2,:), 'V','MarkerEdgeColor','k', 'MarkerFaceColor','g', 'MarkerSize',14), grid;
plotsom(net.iw{1,1},net.layers{1}.distances);
title('����� 20 ������� � �������� ����');
legend('������', '����� ����� ����������', '������ ���������', -1);
hold off;

