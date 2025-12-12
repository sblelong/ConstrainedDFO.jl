export redirect_to_files

function redirect_to_files(dofunc, outfile, errfile)
    return open(outfile, "w") do out
        open(errfile, "w") do err
            redirect_stdout(out) do
                redirect_stderr(err) do
                    dofunc()
                end
            end
        end
    end
end
