function [w] = Hebb_Sigmoid(xtrain, w0, eta)
%% function of Hebbian learning rule for neural networks with single output node
%% return the adjusted weights vector
%% xtrain-row vectors of training data
% number of rows of xtrain represents the number of training sets
% w0-row vector of initialized weights
% eta-learning rate eta

%itr = 0;
%max_itr = 10e6;
w=(w0./norm(w0));
w_diff=100; tol=10e-4;

while(w_diff>tol)
    w_old = w;
    for i = 1:size(xtrain,1)
        net=w*xtrain(i,:)';
        d = tansig(net);
        w = w+eta*d*xtrain(i,:);
    end
    w=w./norm(w);
    w_diff=norm(w_old-w);
end
end