from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


if __name__ == "__main__":
    get_bus_model = BayesianNetwork(
        [
            ('GreenLight', 'GetBus'),
            ('RunToBus', 'GetBus')
        ])

    cpd_green_light = TabularCPD(
        variable='GreenLight',
        variable_card=2,
        values=[[0.67], [0.33]]
    )

    cpd_run_to_bus = TabularCPD(
        variable='RunToBus',
        variable_card=2,
        values=[[0.5], [0.5]]
    )

    cpd_get_bus = TabularCPD(
        variable='GetBus',
        variable_card=2,
        values=[
            [0.88, 0.73, 0.26, 0.05],
            [0.12, 0.27, 0.74, 0.95]
        ],
        evidence=['GreenLight', 'RunToBus'],
        evidence_card=[2, 2],
    )
    get_bus_model.add_cpds(
        cpd_green_light, cpd_run_to_bus, cpd_get_bus
    )

    print(get_bus_model.nodes())
    print(get_bus_model.edges())

    get_bus_infer = VariableElimination(get_bus_model)

    print(get_bus_infer.query(variables=['GreenLight', 'RunToBus', 'GetBus'], evidence={}, show_progress=True)) # E explains away the Alarm

    print(get_bus_model.local_independencies('GetBus'))

    print(get_bus_model.get_independencies())