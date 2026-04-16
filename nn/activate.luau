--!strict
--!optimize 2

local activate = {}

function activate:ReLU()
	for i = 0, (self.size :: number) - 1 do
		local off = self:_offset(i)
		buffer.writef32(self.data, off, math.max(0, buffer.readf32(self.data, off)))
	end
	return self
end

return activate
