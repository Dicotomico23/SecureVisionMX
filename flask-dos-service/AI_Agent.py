import json
from fpdf import FPDF
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from langchain.agents.agent import AgentAction, AgentFinish
from langchain.agents import BaseSingleActionAgent, Tool, AgentExecutor
from langchain_ollama import ChatOllama

app = Flask(__name__)

# Funciones para LangChain que consultan la información del archivo JSON sin ejecutar modelos
def production_analysis(query):
    with open("datos_red.json", "r") as f:
        data = json.load(f)
    return {'data': data["production"]}

def network_analysis(query):
    with open("datos_red.json", "r") as f:
        data = json.load(f)
    return {'data': data["network"]}

def surveillance_analysis(query):
    with open("datos_red.json", "r") as f:
        data = json.load(f)
    return {'data': data["surveillance"]}

# Definición de las herramientas del agente
tools = [
    Tool(
        name="Production Analysis",
        func=production_analysis,
        description="Consulta la información del sistema de producción"
    ),
    Tool(
        name="Network Analysis",
        func=network_analysis,
        description="Consulta la información del sistema de red"
    ),
    Tool(
        name="Surveillance Analysis",
        func=surveillance_analysis,
        description="Consulta la información del sistema de vigilancia"
    )
]

# Generador de reportes PDF que utiliza la información consultada y un LLM para generar análisis en lenguaje natural
def generate_pdf_report(results):
    # Construir prompt para el LLM usando la información del JSON
    analysis_prompt = f"""
Realiza un análisis extenso y completo de la siguiente información de sistemas:

-- PRODUCCIÓN --
{results['production']['data']}

-- RED --
{results['network']['data']}

-- VIGILANCIA --
{results['surveillance']['data']}

Explica en lenguaje natural cada sección: qué información contiene, su relevancia para entender el desempeño operacional y posibles implicaciones o áreas de mejora en el contexto general de la industria. Genera un reporte coherente sin insertar saltos de página innecesarios.
"""
    # Usar Ollama local
    llm = ChatOllama(model="llama3.1")
    response = llm.invoke(analysis_prompt) 
    analysis_text = response.content
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Encabezado del reporte
    pdf.cell(0, 10, txt="Reporte de Análisis Completo de Sistemas", ln=True, align="C")
    pdf.cell(0, 10, txt=f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    
    # Inserción del análisis generado por el LLM en un solo bloque
    pdf.multi_cell(0, 10, analysis_text)
    
    output_filename = "anomaly_report.pdf"
    pdf.output(output_filename)
    return "Reporte generado: " + output_filename

# Agente LangChain que consolida las consultas
class CustomAgent(BaseSingleActionAgent):
    @property
    def input_keys(self):
        return ["input"]

    def plan(self, inputs, **kwargs):
        # Convertir inputs a string si es necesario
        if isinstance(inputs, str):
            inputs = {"input": inputs}
        elif isinstance(inputs, list):
            if len(inputs) == 1:
                inputs = {"input": inputs[0]}
            else:
                inputs = {"input": " ".join(str(item) for item in inputs)}
        
        # Si se consulta por ataques, usar la lógica definida:
        if "ataques" in inputs["input"].lower():
            with open("datos_red.json", "r") as f:
                data = json.load(f)
            count = sum(1 for item in data["network"]["data"] if item.get("status", "").lower() == "advertencia")
            return AgentFinish({'output': f"Hubo {count} ataques hoy."}, log="Consulta de ataques.")

        # Fallback: Usar ChatOllama para consultas generales
        try:
            llm = ChatOllama(model="llama3.1")
            response = llm.invoke(inputs["input"])
            respuesta = response.content
        except Exception as e:
            respuesta = ""
            print("Error al invocar ChatOllama:", e)

        # Si no se obtiene respuesta, asignar mensaje por defecto
        if not respuesta:
            respuesta = "ChatOllama no conectado"
        
        return AgentFinish({'output': respuesta}, log="Respuesta generada por ChatOllama.")

    async def aplan(self, inputs, **kwargs):
        return self.plan(inputs, **kwargs)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=CustomAgent(),
    tools=tools,
    verbose=True
)

@app.route('/api/agent-query', methods=['POST'])
def agent_query():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided."}), 400
    result = agent_executor.invoke(query)

    # Extraer la respuesta del agente; revisar en 'output'
    output = None
    if isinstance(result, dict):
        output = result.get("output")
    else:
        output = result

    if not output:
        output = "ChatOllama no conectado"
        
    return jsonify({'response': output})

@app.route('/')
def dashboard():
    results = {
        'Producción': production_analysis(None),
        'Red': network_analysis(None),
        'Vigilancia': surveillance_analysis(None)
    }
    return render_template('dashboard.html', results=results)

if __name__ == "__main__":
    results = agent_executor.run("Realizar consulta completa de información")
    print(generate_pdf_report(results))