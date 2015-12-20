%% 1.0
clc;
clear;


t = [0 1 0 0 0 1]; % target
x = [-39  45   8  25  0  39; % inputs
      -1 -16 -28 -25 19  45] ./ 10;

net = newp([-5 5; -5 5], 1, 'hardlim', 'learnp'); % creates perceptron
net = configure(net, x, t); % configures network with inputs and target

net.name = 'My Cool Network';

net.inputWeights{1,1}.initFcn = 'rands'; % {numLayers-by-numInputs array}, init function
net.biases{1}.initFcn = 'rands'; % shift

net = init(net);

display(net); % network structure


%plotpv(x,t), grid % plot perceptron input/target vectors
%plotpc(net.IW{1,1}, net.b{1}); % plot classification line


% --- INFO ---
% mae -- Mean absolute error performance function
% mae is a network performance function. 
% It measures network performance as the mean of absolute errors
% ------------


%err = t - y
%h = mae(err);
%disp('h=');
%disp(h);


passes = 9;
[~, columns] = size(x);

% perceptron training
for i = 1 : passes 
    for j = 1 : columns                
        P = x(:, j);
        T = t(:, j);

        IW = net.IW{1, 1}; % array of input weight values
        b = net.b{1}; % array of bias values

        Y = hardlim(IW * P + b);
        E = T - Y;
        perf = mae(E);

        if (~perf)
            continue;
        end

        dW = E * P'; % calculate correction
        db = E; % calculate correction

        net.IW{1, 1} = IW + dW; % correction
        net.b{1} = b + db; % correction
    end

    if (~ mae(t - net(x)))
        break;
    end   
end

y = net(x);

err = t - y;
h = mae(err);
disp({'h=', h});

plotpv(x,t), grid
plotpc(net.IW{1,1}, net.b{1});


%% 1.4

net.inputWeights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net.trainParam.epochs = 50;
net = init(net);
 
[net, tr] = train(net, x, t);

y = net(x);

disp(mae(t - y));

figure;

%plotpv(x,t), grid
%plotpc(net.IW{1,1}, net.b{1});


% random points for testing
lower = min(x, [], 2);
upper = max(x, [], 2);

points = 3;

Tpts = repmat(lower, 1, points) + repmat(upper - lower, 1, points) .* rand(2, points);

disp(Tpts);

TRes = net(Tpts);

%figure;
plotpv(Tpts, TRes);
point = findobj(gca, 'type', 'line');
set(point, 'Color', 'red');
hold on;
plotpv(x ,t);
plotpc(net.IW{1,1}, net.b{1});
grid on
hold off



%% PART 2

disp('new');
%w=[1 0 1 0];
%z=[1 -1 -1 1;
%   1  1 -1 -1];


x = [-39   0   8  25  0  39;
      -1 -16 -28 -25 19  45] ./ 10;


%disp(w);
%disp(z);

net = configure(net, x, t);

net.inputWeights{1, 1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);

%plotpv(x, t), grid;

[net, ty] = train(net, x, t);
figure;
plotpv(x, t), grid
plotpc(net.IW{1, 1}, net.b{1});




%% PART 3
clc;

x1 = [ 39 -46  27 -33 -29  4  -4 -45;
      -41   5 -19 -17   1 12 -11   0];
 
t1 = [0 1 0 0 1 1 0 1;
      1 0 1 0 0 1 0 0];
 
 
net = newp([-5 5; -5 5], 1, 'hardlim', 'learnp');
net = configure(net, x1, t1);

net.inputWeights{1, 1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net.trainParam.epochs = 50;

net = init(net);

[net, tr] = train(net, x1, t1);

figure;
plotpv(x1, t1), grid
plotpc(net.IW{1, 1}, net.b{1});


% random points for testing
lower = min(x1, [], 2);
upper = max(x1, [], 2);

points = 5;

Tpts = repmat(lower, 1, points) + repmat(upper - lower, 1, points) .* rand(2, points);

TRes = net(Tpts);

%figure;
plotpv(Tpts, TRes);
point = findobj(gca, 'type', 'line');
set(point, 'Color', 'red');
hold on;
plotpv(x1 ,t1);
plotpc(net.IW{1,1}, net.b{1});
grid on
hold off

%view(net);
