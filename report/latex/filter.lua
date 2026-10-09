-- Лua-фильтр pandoc: подгоняет таблицы по ширине и размеру шрифта, делает рисунки с подписью «Рис. N», разрешает переносы в коде.
local function utf8len(s) return utf8.len(s) or #s end

local function esc(s)
  s = s:gsub("\\", "\\textbackslash{}")
  s = s:gsub("([{}$&#%%_%^~])", function(c) if c == "^" then return "\\^{}" elseif c == "~" then return "\\~{}" else return "\\" .. c end end)
  return s
end

function Code(el)
  local t = esc(el.text)
  t = t:gsub("([/%.%-=,])", "%1\\allowbreak{}"):gsub("(\\_)", "%1\\allowbreak{}")
  return pandoc.RawInline("latex", "\\texttt{" .. t .. "}")
end

local function unraw(t)   -- Code уже превращён в \texttt{...\allowbreak{}...}: достаём обычный текст
  t = t:gsub("\\allowbreak{}", ""):gsub("^\\texttt{", ""):gsub("}$", ""):gsub("\\([_#%%&{}$])", "%1")
  return t
end

local function cell_text(cell)
  local out = {}
  cell.contents:walk({
    Str = function(x) out[#out + 1] = x.text end,
    Code = function(x) out[#out + 1] = x.text end,
    RawInline = function(x) out[#out + 1] = unraw(x.text) end,
    Math = function(x) out[#out + 1] = x.text end,
    Space = function() out[#out + 1] = " " end,
    SoftBreak = function() out[#out + 1] = " " end,
  })
  return table.concat(out)
end

function Table(tbl)
  local ncols = #tbl.colspecs
  local weights = {}
  for j = 1, ncols do weights[j] = 5 end
  local function visit(rows)
    for _, row in ipairs(rows) do
      for j, cell in ipairs(row.cells) do
        local txt = cell_text(cell)
        local len = utf8len(txt)
        local longest = 0
        for w in txt:gmatch("%S+") do longest = math.max(longest, utf8len(w)) end
        local code = 0
        cell.contents:walk({RawInline = function(c) code = code + utf8len(unraw(c.text)) end})
        weights[j] = math.max(weights[j] or 5, longest + 2, math.min(len + 0.6 * code, 44))
      end
    end
  end
  visit(tbl.head.rows)
  for _, body in ipairs(tbl.bodies) do visit(body.body) end
  local total = 0
  for j = 1, ncols do total = total + weights[j] end
  for j = 1, ncols do tbl.colspecs[j] = {tbl.colspecs[j][1], weights[j] / total} end
  local size = "\\small"
  if ncols >= 9 then size = "\\scriptsize" elseif ncols >= 6 then size = "\\footnotesize" elseif ncols >= 4 then size = "\\small" end
  local nrows = 0
  for _, body in ipairs(tbl.bodies) do nrows = nrows + #body.body end
  local keep = ""
  if nrows <= 6 then keep = "\\Needspace*{" .. math.ceil((nrows + 2) * 1.8) .. "\\baselineskip}" end
  return {pandoc.RawBlock("latex", keep .. "\\begingroup" .. size .. "\\setlength{\\tabcolsep}{4pt}"), tbl, pandoc.RawBlock("latex", "\\endgroup")}
end

function Pandoc(doc)
  local out, blocks, i = {}, doc.blocks, 1
  while i <= #blocks do
    local b = blocks[i]
    local img = nil
    if b.t == "Para" and #b.content == 1 and b.content[1].t == "Image" then img = b.content[1] end
    if b.t == "Figure" then
      b:walk({Image = function(im) img = im end})
    end
    if img then
      local cap = ""
      local nxt = blocks[i + 1]
      if nxt and nxt.t == "Para" and #nxt.content == 1 and nxt.content[1].t == "Emph" then
        cap = pandoc.write(pandoc.Pandoc({pandoc.Plain(nxt.content[1].content)}), "latex")
        i = i + 1
      end
      table.insert(out, pandoc.RawBlock("latex", "\\begin{figure}[H]\\centering\\includegraphics[width=0.94\\linewidth]{" .. img.src .. "}\\caption*{" .. cap .. "}\\end{figure}"))
    else
      table.insert(out, b)
    end
    i = i + 1
  end
  doc.blocks = out
  return doc
end

function HorizontalRule() return {} end
