-- A function to read the relations file and return a table from indices to relation strings
function read_relations(path)
   local path = path
   local rel_file = io.open(path)
   
   local relations_table = {}
   local i = 0
   local line = rel_file:read("*l")
   while line do
      relations_table[i] = line
      i = i+1
      line = rel_file:read("*l")
   end
   return relations_table
end
