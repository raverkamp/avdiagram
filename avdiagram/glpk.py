import subprocess
import tempfile
from typing import List, Optional, Union, Any, Tuple, TextIO, cast

# Wrapper for GPLK as subprocess
# we use the simple input format
# Variables are defined
# the number of constraints

class Var(object):
    def __init__(self,
                 problem: "Problem",
                 int_id:int,
                 key:str,
                 lbound:Optional[float],
                 ubound:Optional[float]):
        self.key  = key
        self.problem = problem
        self.int_id = int_id
        self.lbound = lbound
        self.ubound = ubound
        self.value:float = 0

    def __repr__(self)->str:
        return "Var(key={0},v={1})".format(self.key,self.value)

Term = Tuple[Var,float]
class Constraint(object):
    def __init__(self,int_id:int,l: List[Term],eq_or_le_or_ge:str,bound:float)->None:
        self.l = l
        self.eq_or_le_or_ge = eq_or_le_or_ge
        self.bound = bound
        self.int_id = int_id

class Problem (object):
    def __init__(self)->None:
        self.vars:List[Var] = []
        self.cons:List[Constraint] = []
        self.vardict:dict[str,Var] = {}
        self.objective:List[Term] = []
        self.min_or_max = "Nix"

    def addvar(self,key:str, lbound: Optional[float],ubound: Optional[float])->Var:
        if key in self.vardict:
            raise Exception("Aua")
        v = Var(self, len(self.vars)+1, key, lbound, ubound)
        self.vardict[key] = v
        self.vars.append(v)
        return v

    def addConstraint(self, l:List[Tuple[Var, float]], eq_or_le_or_ge:str, bound:float)->None:
        # l is a list of pairs (var,factor)
        a = []
        #use set here
        seen = {}
        for entry in l:
            (v,fac) = entry
            if v.problem != self:
                raise Exception("Aua")
            if v.key in seen:
                raise Exception("Aua, variables duplicate")
            seen[v.key] = True
            a.append((v,float(fac)))
        if not eq_or_le_or_ge in ['EQ','LE','GE']:
            raise Exception("Aua")
        bound = float(bound)
        c = Constraint(len(self.cons)+1,a, eq_or_le_or_ge,bound)
        self.cons.append(c)

    # v1 = v2 + diff

    def addEQ(self, v1:Var, v2:Var,diff:float=0)->None:
        self.addConstraint([(v1,1),(v2,-1)],'EQ',diff)

    # v1 <= v2 + diff
    def addLE(self,v1:Var,v2:Var,diff:float=0)->None:
        self.addConstraint([(v1,1),(v2,-1)],'LE',diff)

    # v1 => v2 + diff
    def addGE(self,v1: Var,v2:Var,diff:float=0)->None:
        self.addConstraint([(v1,1),(v2,-1)],'GE',diff)

    def addFIX(self,v1:Var,c:float)->None:
        self.addConstraint([(v1,1)],'EQ', c)

    def setObjective(self,min_or_max:str,l:List[Term]) ->None :
        a = []
        #use set here
        seen = {}
        for entry in l:
            (v,fac) = entry
            if v.problem != self:
                raise Exception("Aua")
            if v.key in seen:
                raise Exception("Aua, variables duplicate")
            seen[v.key] = True
            a.append((v,float(fac)))
        self.objective = a
        if min_or_max in ['MIN','MAX']:
            self.min_or_max = min_or_max
        else:
            raise Exception("bad value for min_or_max")


    def countEntries(self)->int :
        n = 0
        for c in self.cons:
            for v in c.l:
                n=n+1
        return n

# eigentlich sollte es anders laufen, das Problem ist unabnhängig vom Solver

    def renderToGLPK(self,out:TextIO)->None:
        if self.min_or_max is None:
            raise "no objective"
        out.write('c Problem\n')
        if self.min_or_max == 'MAX':
            mm = 'max'
        else:
            mm = 'min'
        out.write('p mip {3} {0} {1} {2}\n'.format(len(self.cons),len(self.vars),self.countEntries(),
                                                   mm))
        for c in self.cons:
            i = c.int_id
            out.write('n i {0} cons_{1}\n'.format(i,i))
            if c.eq_or_le_or_ge == 'EQ':
                out.write('i {0} s {1}\n'.format(i,c.bound))
#                out.write('i {0} u {1}\n'.format(i,c.bound))
            elif c.eq_or_le_or_ge == 'LE':
                out.write('i {0} u {1}\n'.format(i,c.bound))
            elif c.eq_or_le_or_ge == 'GE':
                out.write('i {0} l {1}\n'.format(i,c.bound))
            else:
                raise Exception("unknown constraint type")
        for var in self.vars:
            j = var.int_id
            out.write('n j {0} {1}\n'.format(var.int_id, var.key))
            if var.lbound is None and var.ubound is None:
                out.write('j {0} c f\n'.format(var.int_id))
            elif not var.lbound is None and var.ubound is None:
                out.write('j {0} c l {1} \n'.format(var.int_id, var.lbound))
            elif var.lbound is None and not var.ubound is None:
                out.write('j {0} c u {1} \n'.format(var.int_id, var.ubound))
            elif not var.lbound is None and not var.ubound is None and var.lbound < var.ubound:
                out.write('j {0} c d {1} {2} \n'.format(var.int_id, var.lbound, var.ubound))
            elif not var.lbound is None and not var.ubound is None and var.lbound >= var.ubound:
                out.write('j {0} c s {1}\n'.format(var.int_id, var.lbound))
            else:
                raise Exception("BUG")

        # objective
        for entry in self.objective:
            out.write('a 0 {0} {1}\n'.format(entry[0].int_id,entry[1]))
        for c in self.cons:
            for entry in c.l:
                out.write('a {0} {1} {2}\n'.format(c.int_id,entry[0].int_id,entry[1]))
        out.write('e o f\n')


    def parseSolution(self,fileName:str)->List[float]:
        with open(fileName,'r') as input:
            objective:float = 0
            vars: List[float] = []
            for line in input:
                if len(line) == 0:
                    continue
                cc = line[0]
                if cc == 'c':
                    continue
                if cc == 'e':
                    break
                if cc == 's': # solution
                    (_,_,ncons,nvars,uf1,uf2,obj) = line.split(' ')
                    # f == feasible, u == unfeasible?
                    if not (uf1 == 'f' and uf2 == 'f'):
                        return []
                    objective = float(obj)
                    vars = [0.0] * int(nvars)
                if cc == 'j':
                    cols = line.split(' ')
                    j = int(cols[1])
                    v = float(cols[3])
                    vars[j-1] = v
            return vars

    def run(self)->bool:
        for v in self.vars:
            v.value = 0
            with tempfile.NamedTemporaryFile(prefix='glpk-input-', suffix='.txt',mode='w', delete=False) as file_input:
                file_input_name = file_input.name
                self.renderToGLPK(cast(TextIO,file_input))
        with tempfile.NamedTemporaryFile(prefix='glpk-output-', suffix='.txt',mode='w', delete=False) as file_output:
            file_output_name = file_output.name
        subprocess.call(['glpsol','--glp',file_input_name,'--write', file_output_name],
                        stdin=None, stdout=None, stderr=None, shell=False, timeout=None)

        solution_vars = self.parseSolution(file_output_name)
        if len(solution_vars) == 0:
            return False
        for i in range(len(solution_vars)):
            self.vars[i].value = solution_vars[i]
        return True
