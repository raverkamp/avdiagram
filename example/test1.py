from avdiagram import cm, mm, Diagram, Table, Column


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

    d.calign([country, customer_group, customer])

    d.over(customer_group, mm(20), customer)
    d.over(country, mm(20), customer_group)

    d.cline(customer_group, 5, country, 25)
    d.cline(customer, 5, customer_group, 25)

    d.cline(customer_group, 12, customer_group, 8)

    d.show()


main()
