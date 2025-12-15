export redirect_to_files

function redirect_to_files(dofunc, outfile)
    return open(outfile, "w") do out
        redirect_stdout(out) do
            dofunc()
        end
    end
end
