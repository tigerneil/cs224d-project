
local script = {}
function script.setDefault(t, d)
        local mt = {__index = function () return d end}
        setmetatable(t, mt)
end

return script


function readDataFile(path)
    local path = path
    print "in read"
    local inputFile = io.open(path)
    
    local line = inputFile:read("*l")
    while line do
       processLine(line) -- replace this with any function
       line = inputFile:read("*l")
    end
end
