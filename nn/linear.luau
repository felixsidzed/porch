--!strict
--!optimize 2

local Tensor = require("../core/tensor")

type Tensor = Tensor.Tensor

local Linear = {}
Linear.__index = Linear

export type Linear = typeof(setmetatable(
	{} :: {
		weight: Tensor, weightGrad: Tensor,
		bias: Tensor, biasGrad: Tensor,
		input: Tensor?
	},
	Linear
))

function Linear.new(infeat: number, outfeat: number): Linear
	return setmetatable({
		weight = Tensor.randn({infeat, outfeat}):mul(math.sqrt(2 / (infeat + outfeat))),
		bias = Tensor.zeros({1, outfeat}),
		weightGrad = Tensor.zeros({infeat, outfeat}),
		biasGrad = Tensor.zeros({1, outfeat}),
		input = nil
	}, Linear)
end

function Linear:forward(x: Tensor)
	self.input = x
	return x:matmul(self.weight):add(self.bias)
end

function Linear:backward(dout: Tensor)
	local x = self.input
	assert(x ~= nil, "Linear.backward called before forward")

	-- bwo
	local xT = x:transpose(1, 2)
	local dW = xT:matmul(dout)
	
	local dB = (dout:sum(0) :: Tensor):reshape({1, dout.shape[2]})
	
	local wT = self.weight:transpose(1, 2)
	local dX = dout:matmul(wT)

	self.weightGrad:add(dW)
	self.biasGrad:add(dB)

	return dX
end

function Linear:params()
	return { self.weight, self.bias }
end

function Linear:zerograd()
	self.weightGrad:mul(0)
	self.biasGrad:mul(0)
end

return Linear
