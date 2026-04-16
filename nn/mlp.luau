--!strict
--!optimize 2

local Linear = require("./linear")
local ReLU = require("./activate").ReLU
local Tensor = require("../core/tensor")

type Tensor = Tensor.Tensor
type Linear = Linear.Linear

local MLP = {}
MLP.__index = MLP

export type MLP = typeof(setmetatable(
	{} :: {
		layers: {Linear},
		preActs: {Tensor},
		input: Tensor?
	},
	MLP
))

function MLP.new(sizes: {number}): MLP
	local layers = {}

	for i = 1, #sizes - 1 do
		table.insert(layers, Linear.new(sizes[i], sizes[i + 1]))
	end

	return setmetatable({
		layers = layers,
		preActs = {},
		input = nil
	}, MLP)
end

function MLP:forward(x: Tensor): Tensor
	self.input = x
	self.preActs = {}

	for i, layer in self.layers do
		x = layer:forward(x)
		self.preActs[i] = x

		if i < #self.layers then
			ReLU(x)
		end
	end

	return x
end

function MLP:backward(dout: Tensor): Tensor
	local nlayers = #(self :: any).layers
	for i = nlayers, 1, -1 do
		local layer = (self :: any).layers[i]

		if i < nlayers then
			dout:mul(if (self :: any).preActs[i]:gt(0) then 1 else 0)
		end

		dout = layer:backward(dout)
	end

	return dout
end

function MLP:params()
	local params = {}

	for _, layer in self.layers do
		for _, p in layer:params() do
			table.insert(params, p)
		end
	end

	return params
end

function MLP:zerograd()
	for _, layer in self.layers do
		layer:zerograd()
	end
end

return MLP
