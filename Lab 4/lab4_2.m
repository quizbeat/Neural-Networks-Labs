clear;

%карта Кохонена для выполнения кластеризации множества точек

%формирую множество случайных точек

X = [0 1.5; 0 1.5];
clusters = 12; %число кластеров для группировки
points = 10;
deviation = 0.07;
P = nngenc(X, clusters, points, deviation);
figure;
%точки
plot(P(1,:),P(2,:),'ob',...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','c',...
                'MarkerSize',7)
title('120 точек (сгруппированные в 8 кластеров)');
xlabel('Ось X');
ylabel('Ось Y');            
grid on;

%создаю сеть
net = newsom(X,[3 4]);

net.trainParam.epochs = 300; %число эпох обучения
%%
net = train(net,P); %обучаю сеть заданным по умолчанию методом

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

w = net.IW{1,1};  % весовые коэффициенты первого слоя после обучения
display('Весовые коэффициенты первого слоя после обучения:');
display(w);


%Проверяю качество разбиения: случайным образом задаю 5 точек и подаю их в сеть. 

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

%рисую графики
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
hold on;
plotsom(net.iw{1,1},net.layers{1}.distances);

title('Группы из 80 точек и 5 дополнительных случайных точек');
xlabel('Ось X');
ylabel('Ось Y');     
legend('80 исходных точек','Центры кластеризации сети', 'Дополнительные точки', 'Связи между кластерами',  -1);
grid on;

