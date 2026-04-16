--!strict
--!optimize 2

local MLP = require("./nn/mlp")
local Tensor = require("./core/tensor")

local model = MLP.new({3, 8, 2})

local lr = 0.01
local trueW = Tensor.randn({3, 2})

local x = Tensor.randn({4, 3})
local y = x:matmul(trueW)

local start = os.clock()
for step = 1, 100 do
	local pred = model:forward(x)

	local diff = pred - y
	local loss = (diff ^ 2):mean()

	local grad = diff * (2 / diff.size)
	model:zerograd()
	model:backward(grad)

	for _, layers in model.layers do
		layers.weight:sub(layers.weightGrad * lr)
		layers.bias:sub(layers.biasGrad * lr)
	end

	if step % 10 == 0 then
		print("step", step, "loss", loss)
	end
end

print(`{os.clock() - start}ms`)