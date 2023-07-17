from avdiagram import glpk

def a():
    p = glpk.Problem()
    v1 = p.addvar('a',0,10)
    v2 = p.addvar('b',0,10)
    p.addConstraint([(v1,1),(v2,3)],'LE',4)
    p.addConstraint([(v1,12),(v2,3)],'LE',13)
    p.addConstraint([(v1,0.8)],'GE',0.07)
    p.setObjective('MAX',[(v1,1),(v2,1)])
    p.run()
    for k in p.vardict:
        print(k,"=", p.vardict[k].value)

a()
    
    
    
    
        
