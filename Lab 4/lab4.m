%% PART 1   1.1 - 1.4
clear;
clc;

trange = 0 : 0.025 : 2 * pi;

X1 = ellipse(trange, 0.3, 0.3, 0, 0, 0);
P1 = X1(:, randperm(numel(trange), 60));
T1 = [ones(1, 60); zeros(1, 60); zeros(1, 60)];

X2 = ellipse(trange, 0.7, 0.7, 0, 0, 0);
P2 = X2(:, randperm(numel(trange), 100));
T2 = [zeros(1, 100); ones(1, 100); zeros(1, 100)];

X3 = ellipse(trange, 1, 1, 0, 0, 0);
P3 = X3(:, randperm(numel(trange), 120));
T3 = [zeros(1, 120); zeros(1, 120); ones(1, 120)];

[trainInd1,valInd1,testInd1] = dividerand(60, .8, 0, .2);
[trainInd2,valInd2,testInd2] = dividerand(100, .8, 0, .2);
[trainInd3,valInd3,testInd3] = dividerand(120, .8, 0, .2);

figure;
hold on;
plot(X1(1, :), X1(2, :), '-r', 'LineWidth', 2);
plot(P1(1, trainInd1), P1(2, trainInd1), 'or', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
plot(P1(1, testInd1), P1(2, testInd1), 'rs', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c', 'MarkerSize', 7);

plot(X2(1, :), X2(2, :), '-g', 'LineWidth', 2);
plot(P2(1, trainInd2), P2(2, trainInd2), 'og', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g', 'MarkerSize', 7);
plot(P2(1, testInd2), P2(2, testInd2), 'gs', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c', 'MarkerSize', 7);

plot(X3(1, :), X3(2, :), '-b', 'LineWidth', 2);
plot(P3(1, trainInd3), P3(2, trainInd3), 'ob', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
plot(P3(1, testInd3), P3(2, testInd3), 'bs', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'c', 'MarkerSize', 7);

legend('initial set 1', 'train subset 1', 'test subset 1',...
       'initial set 2', 'train subset 2', 'test subset 2',...
       'initial set 3', 'train subset 3', 'test subset 3');
axis([-1.2 1.2 -1.2 1.2]);
grid on;

Ptrain = [P1(:, trainInd1) P2(:, trainInd2) P3(:, trainInd3)];
Ttrain = [T1(:, trainInd1) T2(:, trainInd2) T3(:, trainInd3)];
ntrain = size(Ptrain, 2);

Pval = [P1(:, valInd1) P2(:, valInd2) P3(:, valInd3)];
Tval = [T1(:, valInd1) T2(:, valInd2) T3(:, valInd3)];
nval = size(Pval, 2);

Ptest = [P1(:, testInd1) P2(:, testInd2) P3(:, testInd3)];
Ttest = [T1(:, testInd1) T2(:, testInd2) T3(:, testInd3)];
ntest = size(Ptest, 2);

P = [Ptrain Pval Ptest];
T = [Ttrain Tval Ttest];

%% 1.6 - 1.10
%If spread is near zero, the network acts as a nearest neighbor classifier.
%As spread becomes larger, the designed network takes into account several nearby design vectors.
%radbas spread => biases
%IW = P'
%LW = t

spread = 0.3;
net = newpnn(Ptrain, Ttrain, spread);
view(net);

Ytrain = vec2ind(sim(net, Ptrain));
display(nnz(Ytrain ~= vec2ind(Ttrain))) 

Ytest = vec2ind(sim(net, Ptest));
display(nnz(Ytest ~= vec2ind(Ttest))) 

% PLOT
[X, Y] = meshgrid(-1.2 : 0.025 : 1.2, 1.2 : -0.025 : -1.2);
n = length(X);
out = sim(net, [X(:)'; Y(:)']);
out = max(0, min(1, out));
out = round(out * 10) * 0.1;
ctable = unique(out', 'rows');
cmap = zeros(n, n);
for i = 1 : size(ctable, 1)
    cmap(ismember(out', ctable(i, :), 'rows')) = i; 
end
image([-1.2, 1.2], [-1.2, 1.2], cmap); 
colormap(ctable);

%% 1.11 - 1.12

spread = 0.1;
net = newpnn(Ptrain, Ttrain, spread);
view(net);

% PLOT
[X, Y] = meshgrid(-1.2 : 0.025 : 1.2, 1.2 : -0.025 : -1.2);
n = length(X);
out = sim(net, [X(:)'; Y(:)']);
out = max(0, min(1, out));
out = round(out * 10) * 0.1;
ctable = unique(out', 'rows');
cmap = zeros(n, n);
for i = 1 : size(ctable, 1)
    cmap(ismember(out', ctable(i, :), 'rows')) = i; 
end
image([-1.2, 1.2], [-1.2, 1.2], cmap); 
colormap(ctable);




%% PART 2

%The larger spread is, the smoother the function approximation.
%Too large a spread means a lot of neurons are required to fit a fast-changing function.
%Too small a spread means many neurons are required to fit a smooth function, and the network might not generalize well.
%At each iteration the input vector that results in lowering the network error the most is used to create a radbas neuron.

%% 2.5
spread = 0.3;
goal = 10e-5;
net = newrb(Ptrain, Ttrain, goal, spread);
view(net);

%% 2.7 - 2.9

Ytrain = vec2ind(sim(net, Ptrain));
display(nnz(Ytrain ~= vec2ind(Ttrain))) 

Ytest = vec2ind(sim(net, Ptest));
display(nnz(Ytest ~= vec2ind(Ttest))) 

% PLOT
[X, Y] = meshgrid(-1.2 : 0.025 : 1.2, 1.2 : -0.025 : -1.2);
n = length(X);
out = sim(net, [X(:)'; Y(:)']);
out = max(0, min(1, out));
out = round(out * 10) * 0.1;
ctable = unique(out', 'rows');
cmap = zeros(n, n);
for i = 1 : size(ctable, 1)
    cmap(ismember(out', ctable(i, :), 'rows')) = i; 
end
image([-1.2, 1.2], [-1.2, 1.2], cmap); 
colormap(ctable);

%% 2.10 - 2.11

spread = 0.1;
goal = 10e-5;
net = newrb(Ptrain, Ttrain, goal, spread);
view(net);

% PLOT
[X, Y] = meshgrid(-1.2 : 0.025 : 1.2, 1.2 : -0.025 : -1.2);
n = length(X);
out = sim(net, [X(:)'; Y(:)']);
out = max(0, min(1, out));
out = round(out * 10) * 0.1;
ctable = unique(out', 'rows');
cmap = zeros(n, n);
for i = 1 : size(ctable, 1)
    cmap(ismember(out', ctable(i, :), 'rows')) = i; 
end
image([-1.2, 1.2], [-1.2, 1.2], cmap); 
colormap(ctable);




%% PART 3

t0 = 0;
tn = 4;
dt = 0.02;
n = (tn - t0) / dt + 1;
func = @(t) cos(t .^ 2);
x = func(t0 : dt : tn);

trainInd = 1 : floor(n * 0.9);
valInd = [];
testInd = floor(n * 0.9) + 1 : n;
%[trainInd, valInd, testInd] = dividerand(n, .8, 0, .2);

net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

spread = dt;
time = t0 : dt : tn;
%The first layer has as many neurons as there are input/ target vectors in P.
%Specifically, the first-layer weights are set to P'.
%spread => bias
%The second layer also has as many neurons as input/target vectors, but here LW{2,1} is set to T.
net = newgrnn(time(trainInd), x(trainInd), spread);
view(net)


%% 3.5 - 3.6
y = sim(net, t0 : dt : tn);

sqrt(mse(x(trainInd) - y(trainInd)))
sqrt(mse(x(testInd) - y(testInd)))

figure;
hold on;
plot(t0 : dt : tn, x, '-b');
plot(t0 : dt : tn,  y, '-r');
grid on;

figure;
plot(t0 : dt : tn, x - y);
grid on;

%% 3.7 - 3.8

[trainInd, valInd, testInd] = dividerand(n, .8, 0, .2);

net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

spread = dt;
time = t0 : dt : tn;

net = newgrnn(time(trainInd), x(trainInd), spread);
view(net)

y = sim(net, t0 : dt : tn);

sqrt(mse(x(trainInd) - y(trainInd)))
sqrt(mse(x(testInd) - y(testInd)))

figure;
hold on;
plot(t0 : dt : tn, x, '-b');
plot(t0 : dt : tn,  y, '-r');
grid on;

figure;
plot(t0 : dt : tn, x - y);
grid on;

