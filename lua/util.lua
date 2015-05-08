
local script = {}
function script.setDefault(t, d)
        local mt = {__index = function () return d end}
        setmetatable(t, mt)
end

return script
