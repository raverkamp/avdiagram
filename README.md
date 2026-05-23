# Layout Diagrams with linear Constraints

This is an experiment to layout diagrams with linear constraints.

The creation of a diagram is done with python function calls. There is no DSL.
The constraints are translated into a linear program and then solved by an external solver. 

A simple ERD:
``` 
def main():
    d = Diagram(cm(20), cm(20))

    customer = d.table(
        Table(
            "CUSTOMER",
            [
                Column("ID", "Integer"),
                Column("CUSTOMER_NUMBER", "String"),
                Column("MASTER_CUSTOMER_ID", "Integer"),
                Column("Address ...", "String"),
            ],
            None,
        )
    )
    customer_group = d.table(
        Table(
            "CUSTOMER_GROUP",
            [Column("ID", "Integer"), Column("KIND", "String"), Column("...", "...")],
            "ID",
        )
    )

    country = d.table(
        Table(
            "COUNTRY",
            [
                Column("ID", "Integer"),
                Column("ISO_CODE", "String"),
                Column("NAME", "String"),
            ],
            "ID",
        )
    )

    # align tables by there horizontal center
    d.calign([country, customer_group, customer])

    # stack the tables with a vertical distance of 20mm
    d.over(customer_group, mm(20), customer)
    d.over(country, mm(20), customer_group)

    connect(customer_group, 5, country, 25)
    connect(customer, 5, customer_group, 25)

    connect(customer_group, 12, customer_group, 8)

    d.show()
``` 


produces

<image src="./ERD-1.png" alt="ERD">


A more complex example 
[example/dwh_process.py](example/dwh_process.py)

<image src="./dwh.png" alt="dwh">