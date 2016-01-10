clear;

%слой Кохонена для выполнения кластеризации множества точек

%формирую множество случайных точек
X = [0 1.5; 0 1.5];
clusters = 8;
points = 10;
deviation = 0.01;
P = nngenc(X, clusters, points, deviation);
figure;
% 80 точек, которые изначально сгруппированы в 8 кластеров
plot(P(1,:),P(2,:),'ob',...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','c',...
                'MarkerSize',7)
title('80 точек (сгруппированные в 8 кластеров)');
xlabel('Ось X');
ylabel('Ось Y');            
grid on;
%%
net = competlayer(8); %создаю сеть
net = configure(net, P);

net.trainParam.epochs = 50;
net = train(net,P);

display(net);
view(net);
%отображаю структуру сети и проведенное обучение
display('Метод инициализации сети:');display(net.initFcn);
display('Активационная функция скрытого слоя:');
display(net.layers{1}.transferFcn);
display('Число нейронов в скрытом слое:');display(net.layers{1}.size);
%параметры обучения:
display(net.trainParam.epochs);
display(net.trainParam.show);
display(net.trainParam.showCommandLine);
display(net.trainParam.showWindow);
display(net.trainParam.time);
%%
w = net.IW{1,1};  % центры кластеризации как результирующие веса после обучения

display('Центры кластеризации сети:');
display(w);

%задаю 5 случайных двухмерных входных векторов
Pdop5(1:2, 1:5)= 0;
for i = 1 : 2
    for j = 1 : 5
        Pdop5(i,j)=rand; 
    end
end

display('Пять дополнительных точек:');
display(Pdop5);
display('Кластеры пяти дополнительных точек:');
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
plot(Pdop5(1,:),Pdop5(2,:),'rs',...                %визуальное изображение 5 дополнительных случайных точек
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','y',...
                'MarkerSize',7);    
title('Группы из 80 точек и 5 дополнительных случайных точек');
xlabel('Ось X');
ylabel('Ось Y');     
legend('80 исходных точек','Центры кластеризации сети', 'Дополнительные точки', -1);
grid on;

