--!strict
--!optimize 2

local Linear = require("./nn/linear")
local Tensor = require("./core/tensor")

local layer = Linear.new(3, 2)

local lr = 0.01
local trueW = Tensor.randn({3, 2})

local x = Tensor.randn({4, 3})
local y = x:matmul(trueW)

local start = os.clock()
for step = 1, 100 do
	-- flushed ref
	local pred = layer:forward(x)

	local diff = pred - y
	local loss = (diff ^ 2):mean()

	local grad = diff * (2 / diff.size)
	layer:zerograd()
	layer:backward(grad)

	layer.weight:sub(layer.weightGrad * lr)
	layer.bias:sub(layer.biasGrad * lr)

	if step % 10 == 0 then
		print("step", step, "loss", loss)
	end
end
print(`{os.clock() - start}ms`)
