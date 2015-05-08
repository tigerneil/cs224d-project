--LOAD WORD VECTORS

--Load custom word2vec word vectors
function load_word2vec(w2vpath, inputDim)
    local path = w2vpath
    local inputDim = inputDim
    print('Using custom word vectors')
    local ignore = io.open(opt.wordTable,"r")
    if ignore ~= nil then
      io.close(ignore)
      return torch.load(opt.wordTable)
    end
    local word2vec_file = io.open(path)
    local word2vec_table = {}

    local line = word2vec_file:read("*l")
    while line do
        -- read the word2vec text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    word2vec_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                word2vec_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = word2vec_file:read("*l")
    end
    print('Saving dictionary as torch file for later')
    torch.save(opt.wordTable, word2vec_table)
    return word2vec_table
end

--Load word vectors using glove or w2v
function load_wordVector(wordVectorPath, inputDim, model)
    print("Start loading word vectors")
    local inputDim = inputDim
    local path = wordVectorPath
    if model == 'wv' then
      return load_word2vec(wordVectorPaht, inputDim)
    end
    local wordVector_file = io.open(path)
    local wordVector_table = {}

    local line = wordVector_file:read("*l")
    while line do
        -- read the wordVector text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    wordVector_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                wordVector_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = wordVector_file:read("*l")
    end
    
    return wordVector_table
end
